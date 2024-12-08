fn main() {
    println!("Testing non erroneous option_take_on_temporary");
    let mut option = Some(1);
    let _ = Box::new(move || option.take().unwrap());

    println!("Testing non erroneous option_take_on_temporary");
    let x = Some(3);
    x.as_ref();

    println!("Testing erroneous option_take_on_temporary");
    let x = Some(3);
    x.as_ref().take();
}
