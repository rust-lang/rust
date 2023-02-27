// run-pass
fn test1() {
    let i = 0;
    let c = i + --i;
    println!("{c}");
}
fn test2() {
    let i = 9;
    let c = -- i + --i;
    println!("{c}");
}

fn test3(){
    let i=10;
    println!("{}",i--i);
}
fn test4(){
    let i=10;
    println!("{}",--i);

}
struct Foo {
    bar: Bar,
}

struct Bar {
    qux: i32,
}

fn test5() {
    let foo = Foo { bar: Bar { qux: 0 } };
    let c=--foo.bar.qux;
    println!("{c}");
}

fn test6(){
    let x=2;
    let y=--x;
    println!("{y}");
}
fn main(){
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
}
