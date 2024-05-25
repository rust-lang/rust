use std::panic;

fn main(){
    panic::highlight_errors(true);
    //~^ ERROR E0658
}
