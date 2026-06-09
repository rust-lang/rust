struct Test;

struct Test2 {
    b: Option<Test>,
}

struct Test3(Option<Test>);

impl Drop for Test {
    fn drop(&mut self) {
        println!("dropping!");
    }
}

impl Drop for Test2 {
    fn drop(&mut self) {}
}

impl Drop for Test3 {
    fn drop(&mut self) {}
}

fn stuff() {
    let mut t = Test2 { b: None };
    let u = Test;
    drop(t);
    t.b = Some(u);
    //~^ ERROR assign of moved value: `t`

    let mut t = Test3(None);
    let u = Test;
    drop(t);
    t.0 = Some(u);
    //~^ ERROR assign of moved value: `t`
}

fn main() {
    stuff()
}
