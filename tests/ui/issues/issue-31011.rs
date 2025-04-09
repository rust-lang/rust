macro_rules! log {
    ( $ctx:expr, $( $args:expr),* ) => {
        if $ctx.trace {
        //~^ ERROR no field `trace` on type `&T`
            println!( $( $args, )* );
        }
    }
}

// Create a structure.
struct Foo {
  trace: bool,
}

// Generic wrapper calls log! with a structure.
fn wrap<T>(context: &T) -> ()
{
    log!(context, "entered wrapper");
    //~^ in this expansion of log!
}

fn main() {
    // Create a structure.
    let x = Foo { trace: true };
    log!(x, "run started");
    // Apply a closure which accesses internal fields.
    wrap(&x);
    log!(x, "run finished");
}
