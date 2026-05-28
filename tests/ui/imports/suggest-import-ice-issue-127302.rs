//@ revisions: edition2015 edition2021

mod config; //~ ERROR file not found for module

fn main() {
    match &args.cmd { //~ ERROR cannot find value `args` in this scope
        crate::config => {} //~ ERROR expected unit struct, unit variant or constant, found module `crate::config`
    }

    println!(args.ctx.compiler.display());
    //~^ ERROR format argument must be a string literal
}
