struct MyStruct;

impl MyStruct {
    pub fn get_option() -> Option<Self> {
        todo!()
    }
}

fn return_option() -> Option<i32> {
    todo!()
}

fn main() {
    println!("Testing non erroneous option_take_on_temporary");
    let mut option = Some(1);
    let _ = Box::new(move || option.take().unwrap());

    println!("Testing non erroneous option_take_on_temporary");
    let x = Some(3);
    x.as_ref();

    let x = Some(3);
    x.as_ref().take();
    //~^ ERROR: called `Option::take()` on a temporary value

    println!("Testing non erroneous option_take_on_temporary");
    let mut x = Some(3);
    let y = x.as_mut();

    let mut x = Some(3);
    let y = x.as_mut().take();
    //~^ ERROR: called `Option::take()` on a temporary value
    let y = x.replace(289).take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let y = Some(3).as_mut().take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let y = Option::as_mut(&mut x).take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let x = return_option();
    let x = return_option().take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let x = MyStruct::get_option();
    let x = MyStruct::get_option().take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let mut my_vec = vec![1, 2, 3];
    my_vec.push(4);
    let y = my_vec.first();
    let y = my_vec.first().take();
    //~^ ERROR: called `Option::take()` on a temporary value

    let y = my_vec.first().take();
    //~^ ERROR: called `Option::take()` on a temporary value
}
