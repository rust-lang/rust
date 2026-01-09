// This checks that incorrect params on function parameters are caught

fn function(#[inline] param: u32) {
    //~^ ERROR attribute cannot be used on
}

fn main() {}
