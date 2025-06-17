// Make sure we don't suggest compatible variants inside macro. (issue #142359)
use std::ops::ControlFlow;

fn main(){
    let x: Result<i32, i32> = Err(1);

    let v= match x {
        Err(r) => ControlFlow::Break(r),
        Ok(r) => { println!("A")} //~ ERROR `match` arms have incompatible types [E0308]
    };
}
