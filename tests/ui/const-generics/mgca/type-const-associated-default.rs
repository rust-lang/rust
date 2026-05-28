#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
trait Trait {
    type const N: usize = 10;
    //~^ ERROR associated type defaults are unstable
}

fn main(){
}
