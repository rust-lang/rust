#[derive(Send)]
//~^ ERROR this unsafe trait should be implemented explicitly
struct Test;

#[derive(Sync)]
//~^ ERROR this unsafe trait should be implemented explicitly
struct Test1;

pub fn main() {}
