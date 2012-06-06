export swappable;
export unwrap;
export methods;

#[doc = "
A value that may be swapped out temporarily while it is being processed
and then replaced.  Swappables are most useful when working with unique
values, which often cannot be mutated unless they are stored in the local
stack frame to ensure memory safety.

The type guarantees the invariant that the value is always \"swapped in\"
except during the execution of the `swap()` and `with()` methods.
"]
type swappable<A> = {
    mut o_t: option<A>
};

#[doc = "Create a swappable swapped in with a given initial value"]
fn swappable<A>(+t: A) -> swappable<A> {
    {mut o_t: some(t)}
}

#[doc = "Consumes a swappable and returns its contents without copying"]
fn unwrap<A>(-s: swappable<A>) -> A {
    let {o_t: o_t} <- s;
    option::unwrap(o_t)
}

impl methods<A> for swappable<A> {
    #[doc = "
         Overwrites the contents of the swappable
    "]
    fn set(+a: A) {
        self.o_t <- some(a);
    }

    #[doc = "
         Invokes `f()` with the current value but replaces the
         current value when complete.  Returns the result of `f()`.

         Attempts to read or access the receiver while `f()` is executing
         will fail dynamically.
    "]
    fn with<B>(f: fn(A) -> B) -> B {
        let mut o_u = none;
        self.swap { |t| o_u <- some(f(t)); t }
        option::unwrap(o_u)
    }

    #[doc = "
         Invokes `f()` with the current value and then replaces the
         current value with the result of `f()`.

         Attempts to read or access the receiver while `f()` is executing
         will fail dynamically.
    "]
    fn swap(f: fn(-A) -> A) {
        alt self.o_t {
          none { fail "no value present---already swapped?"; }
          some(_) {}
        }

        let mut o_t = none;
        o_t <-> self.o_t;
        self.o_t <- some(f(option::unwrap(o_t)));
    }

    #[doc = "True if there is a value present in this swappable"]
    fn is_present() -> bool {
        alt self.o_t {
          none {false}
          some(_) {true}
        }
    }

    #[doc = "
        Removes the value from the swappable.  Any further attempts
        to use the swapabble without first invoking `set()` will fail.
    "]
    fn take() -> A {
        alt self.o_t {
          none { fail "swapped out"; }
          some(_) {}
        }

        let mut o_t = none;
        option::unwrap(o_t)
    }
}

impl methods<A:copy> for swappable<A> {
    #[doc = "
        Copies out the contents of the swappable
    "]
    fn get() -> A {
        self.o_t.get()
    }
}