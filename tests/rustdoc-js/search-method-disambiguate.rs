pub trait X {
    type InnerType;
    fn my_method(&self) -> Self::InnerType;
}

pub struct MyTy<T> {
    pub t: T,
}

impl X for MyTy<bool> {
    type InnerType = bool;
    fn my_method(&self) -> bool {
        self.t
    }
}

impl X for MyTy<u8> {
    type InnerType = u8;
    fn my_method(&self) -> u8 {
        self.t
    }
}
