// run-rustfix

fn main() {
    println!("Testing option_take_on_temporary");
    let x = Some(3);
    x.as_ref().take();
}
