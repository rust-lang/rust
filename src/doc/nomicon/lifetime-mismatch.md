% Limits of Lifetimes

Given the following code:

```rust,ignore
struct Foo;

impl Foo {
    fn mutate_and_share(&mut self) -> &Self { &*self }
    fn share(&self) {}
}

fn main() {
    let mut foo = Foo;
    let loan = foo.mutate_and_share();
    foo.share();
}
```

One might expect it to compile. We call `mutate_and_share`, which mutably borrows
`foo` temporarily, but then returns only a shared reference. Therefore we
would expect `foo.share()` to succeed as `foo` shouldn't be mutably borrowed.

However when we try to compile it:

```text
<anon>:11:5: 11:8 error: cannot borrow `foo` as immutable because it is also borrowed as mutable
<anon>:11     foo.share();
              ^~~
<anon>:10:16: 10:19 note: previous borrow of `foo` occurs here; the mutable borrow prevents subsequent moves, borrows, or modification of `foo` until the borrow ends
<anon>:10     let loan = foo.mutate_and_share();
                         ^~~
<anon>:12:2: 12:2 note: previous borrow ends here
<anon>:8 fn main() {
<anon>:9     let mut foo = Foo;
<anon>:10     let loan = foo.mutate_and_share();
<anon>:11     foo.share();
<anon>:12 }
          ^
```

What happened? Well, we got the exact same reasoning as we did for
[Example 2 in the previous section][ex2]. We desugar the program and we get
the following:

```rust,ignore
struct Foo;

impl Foo {
    fn mutate_and_share<'a>(&'a mut self) -> &'a Self { &'a *self }
    fn share<'a>(&'a self) {}
}

fn main() {
	'b: {
    	let mut foo: Foo = Foo;
    	'c: {
    		let loan: &'c Foo = Foo::mutate_and_share::<'c>(&'c mut foo);
    		'd: {
    			Foo::share::<'d>(&'d foo);
    		}
    	}
    }
}
```

The lifetime system is forced to extend the `&mut foo` to have lifetime `'c`,
due to the lifetime of `loan` and mutate_and_share's signature. Then when we
try to call `share`, and it sees we're trying to alias that `&'c mut foo` and
blows up in our face!

This program is clearly correct according to the reference semantics we actually
care about, but the lifetime system is too coarse-grained to handle that.


TODO: other common problems? SEME regions stuff, mostly?




[ex2]: lifetimes.html#example-aliasing-a-mutable-reference
