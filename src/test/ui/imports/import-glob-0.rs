use module_of_many_things::*;

mod module_of_many_things {
    pub fn f1() { println!("f1"); }
    pub fn f2() { println!("f2"); }
    fn f3() { println!("f3"); }
    pub fn f4() { println!("f4"); }
}


fn main() {
    f1();
    f2();
    f999(); //~ ERROR cannot find function `f999` in this scope
    f4();
}
