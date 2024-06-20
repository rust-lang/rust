pub trait SomeTrait {}
pub trait OtherThingxxxxxxxx {}

pub fn alef<T: OtherThingxxxxxxxx>() -> Result<T, ()> {
    loop {}
}
pub fn bet<T: SomeTrait>() -> Result<T, ()> {
    loop {}
}

pub fn alpha<T: OtherThingxxxxxxxx>(_param: Result<T, ()>) {
    loop {}
}
pub fn beta<T: SomeTrait>(_param: Result<T, ()>) {
    loop {}
}
