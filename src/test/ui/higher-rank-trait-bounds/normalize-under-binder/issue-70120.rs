// check-pass

pub trait MyTrait<'a> {
    type Output: 'a;
    fn gimme_value(&self) -> Self::Output;
}

pub struct MyStruct;

impl<'a> MyTrait<'a> for MyStruct {
    type Output = &'a usize;
    fn gimme_value(&self) -> Self::Output {
        unimplemented!()
    }
}

fn meow<T, F>(t: T, f: F)
where
    T: for<'any> MyTrait<'any>,
    F: for<'any2> Fn(<T as MyTrait<'any2>>::Output),
{
    let v = t.gimme_value();
    f(v);
}

fn main() {
    let struc = MyStruct;
    meow(struc, |foo| {
        println!("{:?}", foo);
    })
}
