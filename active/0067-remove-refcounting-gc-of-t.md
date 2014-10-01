- Start Date: 2014-09-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/256
- Rust Issue: https://github.com/rust-lang/rfcs/pull/256

# Summary

Remove the reference-counting based `Gc<T>` type from the standard
library and its associated support infrastructure from `rustc`.

Doing so lays a cleaner foundation upon which to prototype a proper
tracing GC, and will avoid people getting incorrect impressions of
Rust based on the current reference-counting implementation.

# Motivation

## Ancient History

Long ago, the Rust language had integrated support for automatically
managed memory with arbitrary graph structure (notably, multiple
references to the same object), via the type constructors `@T` and
`@mut T` for any `T`.  The intention was that Rust would provide a
task-local garbage collector as part of the standard runtime for Rust
programs.

As a short-term convenience, `@T` and `@mut T` were implemented via
reference-counting: each instance of `@T`/`@mut T` had a reference
count added to it (as well as other meta-data that were again for
implementation convenience).  To support this, the `rustc` compiler
would emit, for any instruction copying or overwriting an instance of
`@T`/`@mut T`, code to update the reference count(s) accordingly.

(At the same time, `@T` was still considered an instance of `Copy` by
the compiler.  Maintaining the reference counts of `@T` means that you
*cannot* create copies of a given type implementing `Copy` by
`memcpy`'ing blindly; one must distinguish so-called "POD" data that
is `Copy and contains no `@T` from "non-POD" `Copy` data that can
contain `@T` and thus must be sure to update reference counts when
creating a copy.)

Over time, `@T` was replaced with the library type `Gc<T>` (and `@mut
T` was rewritten as `Gc<RefCell<T>>`), but the intention was that Rust
would still have integrated support for a garbage collection.  To
continue supporting the reference-count updating semantics, the
`Gc<T>` type has a lang item, `"gc"`.  In effect, all of the compiler
support for maintaining the reference-counts from the prior `@T` was
still in place; the move to a library type `Gc<T>` was just a shift in
perspective from the end-user's point of view (and that of the
parser).

## Recent history: Removing uses of Gc<T> from the compiler

Largely due to the tireless efforts of `eddyb`, one of the primary
clients of `Gc<T>`, namely the `rustc` compiler itself, has little to
no remaining uses of `Gc<T>`.

## A new hope

This means that we have an opportunity now, to remove the `Gc<T>` type
from `libstd`, and its associated built-in reference-counting support
from `rustc` itself.

I want to distinguish removal of the particular reference counting
`Gc<T>` from our compiler and standard library (which is what is being
proposed here), from removing the goal of supporting a garbage
collected `Gc<T>` in the future. I (and I think the majority of the
Rust core team) still believe that there are use cases that would be
well handled by a proper tracing garbage collector.

The expected outcome of removing reference-counting `Gc<T>` are as follows:

 * A cleaner compiler code base,

 * A cleaner standard library, where `Copy` data can be indeed copied
    blindly (assuming the source and target types are in agreement,
    which is required for a tracing GC),

 * It would become impossible for users to use `Gc<T>` and then get
   incorrect impressions about how Rust's GC would behave in the
   future.  In particular, if we leave the reference-counting `Gc<T>`
   in place, then users may end up depending on implementation
   artifacts that we would be pressured to continue supporting in the
   future.  (Note that `Gc<T>` is already marked "experimental", so
   this particular motivation is not very strong.)

# Detailed design

Remove the `std::gc` module.  This, I believe, is the extent of the
end-user visible changes proposed by this RFC, at least for users who
are using `libstd` (as opposed to implementing their own).

Then remove the `rustc` support for `Gc<T>`. As part of this, we can
either leave in or remove the `"gc"` and `"managed_heap"` entries in
the lang items table (in case they could be of use for a future GC
implementation).  I propose leaving them, but it does not matter
terribly to me.  The important thing is that once `std::gc` is gone,
then we can remove the support code associated with those two lang
items, which is the important thing.

# Drawbacks

Taking out the reference-counting `Gc<T>` now may lead people to think
that Rust will never have a `Gc<T>`.

 * In particular, having `Gc<T>` in place now means that it is easier
   to argue for putting in a tracing collector (since it would be a
   net win over the status quo, assuming it works).

   (This sub-bullet is a bit of a straw man argument, as I suspect any
   community resistance to adding a tracing GC will probably be
   unaffected by the presence or absence of the reference-counting
   `Gc<T>`.)

 * As another related note, it may confuse people to take out a
   `Gc<T>` type now only to add another implementation with the same
   name later.  (Of course, is that more or less confusing than just
   replacing the underlying implementation in such a severe manner.)

Users may be using `Gc<T>` today, and they would have to switch to
some other option (such as `Rc<T>`, though note that the two are not
100% equivalent; see [Gc versus Rc] appendix).

# Alternatives

Keep the `Gc<T>` implementation that we have today, and wait until we
have a tracing GC implemented and ready to be deployed before removing
the reference-counting infrastructure that had been put in to support
`@T`.  (Which may never happen, since adding a tracing GC is only a
goal, not a certainty, and thus we may be stuck supporting the
reference-counting `Gc<T>` until we eventually do decide to remove
`Gc<T>` in the future.  So this RFC is just suggesting we be proactive
and pull that band-aid off now.

# Unresolved questions

None yet.

# Appendices

## Gc versus Rc

There are performance differences between the current ref-counting
`Gc<T>` and the library type `Rc<T>`, but such differences are beneath
the level of abstraction of interest to this RFC.  The main user
observable difference between the ref-counting `Gc<T>` and the library
type `Rc<T>` is that cyclic structure allocated via `Gc<T>` will be
torn down when the task itself terminates successfully or via unwind.

The following program illustrates this difference.  If you have a
program that is using `Gc` and is relying on this tear-down behavior
at task death, then switching to `Rc` will not suffice.

```rust
use std::cell::RefCell;
use std::gc::{GC,Gc};
use std::io::timer;
use std::rc::Rc;
use std::time::Duration;

struct AnnounceDrop { name: String }

#[allow(non_snake_case)]
fn AnnounceDrop<S:Str>(s:S) -> AnnounceDrop {
    AnnounceDrop { name: s.as_slice().to_string() }
}

impl Drop for AnnounceDrop{ 
    fn drop(&mut self) {
       println!("dropping {}", self.name);
    }
}

struct RcCyclic<D> { _on_drop: D, recur: Option<Rc<RefCell<RcCyclic<D>>>> }
struct GcCyclic<D> { _on_drop: D, recur: Option<Gc<RefCell<GcCyclic<D>>>> }

type RRRcell<D> = Rc<RefCell<RcCyclic<D>>>;
type GRRcell<D> = Gc<RefCell<GcCyclic<D>>>;

fn make_rc_and_gc<S:Str>(name: S) -> (RRRcell<AnnounceDrop>, GRRcell<AnnounceDrop>) {
    let name = name.as_slice().to_string();
    let rc_cyclic = Rc::new(RefCell::new(RcCyclic {
        _on_drop: AnnounceDrop(name.clone().append("-rc")),
        recur: None,
    }));

    let gc_cyclic = box (GC) RefCell::new(GcCyclic {
        _on_drop: AnnounceDrop(name.append("-gc")),
        recur: None,
    });

    (rc_cyclic, gc_cyclic)
}

fn make_proc(name: &str, sleep_time: i64, and_then: proc():Send) -> proc():Send {
    let name = name.to_string();
    proc() {
        let (rc_cyclic, gc_cyclic) = make_rc_and_gc(name);

        rc_cyclic.borrow_mut().recur = Some(rc_cyclic.clone());
        gc_cyclic.borrow_mut().recur = Some(gc_cyclic);

        timer::sleep(Duration::seconds(sleep_time));

        and_then();
    }
}

fn main() {
    let (_rc_noncyclic, _gc_noncyclic) = make_rc_and_gc("main-noncyclic");

    spawn(make_proc("success-cyclic", 2, proc () {}));

    spawn(make_proc("failure-cyclic", 1, proc () { fail!("Oop"); }));

    println!("Hello, world!")
}
```

The above program produces output as follows:

```
% rustc gc-vs-rc-sample.rs && ./gc-vs-rc-sample
Hello, world!
dropping main-noncyclic-gc
dropping main-noncyclic-rc
task '<unnamed>' failed at 'Oop', gc-vs-rc-sample.rs:60
dropping failure-cyclic-gc
dropping success-cyclic-gc
```

This illustrates that both `Gc<T>` and `Rc<T>` will be reclaimed when
used to represent non-cyclic data (the cases labelled
`main-noncyclic-gc` and `main-noncyclic-rc`. But when you actually
complete the cyclic structure, then in the tasks that run to
completion (either successfully or unwinding from a failure), we still
manage to drop the `Gc<T>` cyclic structures, illustrated by the
printouts from the cases labelled `failure-cyclic-gc` and
`success-cyclic-gc`.
