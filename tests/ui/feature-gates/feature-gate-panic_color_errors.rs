use std::panic;

fn main(){
    panic::highlight_errors(Some(true));
    //~^ ERROR E0658
}
