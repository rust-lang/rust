// Test related to when a region bound is required to be specified.

trait IsStatic : 'static { }
trait IsSend : Send { }
trait Is<'a> : 'a { }
trait Is2<'a> : 'a { }
trait SomeTrait { }

// Bounds on object types:

struct Foo<'a,'b,'c> { //~ ERROR parameter `'c` is never used
    // All of these are ok, because we can derive exactly one bound:
    a: Box<IsStatic>,
    b: Box<Is<'static>>,
    c: Box<Is<'a>>,
    d: Box<IsSend>,
    e: Box<Is<'a>+Send>, // we can derive two bounds, but one is 'static, so ok
    f: Box<SomeTrait>,   // OK, defaults to 'static due to RFC 599.
    g: Box<SomeTrait+'a>,

    z: Box<Is<'a>+'b+'c>,
    //~^ ERROR only a single explicit lifetime bound is permitted
    //~| ERROR lifetime bound not satisfied
}

fn test<
    'a,
    'b,
    A:IsStatic,
    B:Is<'a>+Is2<'b>, // OK in a parameter, but not an object type.
    C:'b+Is<'a>+Is2<'b>,
    D:Is<'a>+Is2<'static>,
    E:'a+'b           // OK in a parameter, but not an object type.
>() { }

fn main() { }
