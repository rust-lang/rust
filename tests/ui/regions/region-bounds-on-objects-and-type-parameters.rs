// Test related to when a region bound is required to be specified.

trait IsStatic : 'static { }
trait IsSend : Send { }
trait Is<'a> : 'a { }
trait Is2<'a> : 'a { }
trait SomeTrait { }

// Bounds on object types:

struct Foo<'a,'b,'c> { //~ ERROR parameter `'c` is never used
    // All of these are ok, because we can derive exactly one bound:
    a: Box<dyn IsStatic>,
    b: Box<dyn Is<'static>>,
    c: Box<dyn Is<'a>>,
    d: Box<dyn IsSend>,
    e: Box<dyn Is<'a>+Send>, // we can derive two bounds, but one is 'static, so ok
    f: Box<dyn SomeTrait>,   // OK, defaults to 'static due to RFC 599.
    g: Box<dyn SomeTrait+'a>,

    z: Box<dyn Is<'a>+'b+'c>,
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
