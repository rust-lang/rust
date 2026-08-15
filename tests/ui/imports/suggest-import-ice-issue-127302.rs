//@ revisions: edition2015 edition2021

mod config; //~ ERROR file not found for module

fn main() {
    match &args.cmd { //~ ERROR cannot find value `args` in this scope
        crate::config => {} //~ ERROR cannot find unit struct, unit variant or constant `config` in the crate root
    }

    println!(args.ctx.compiler.display());
    //~^ ERROR format argument must be a string literal
}
