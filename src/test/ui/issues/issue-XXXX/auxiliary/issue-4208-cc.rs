#![crate_name="numeric"]
#![crate_type = "lib"]

pub trait Trig<T> {
    fn sin(&self) -> T;
}

pub fn sin<T:Trig<R>, R>(theta: &T) -> R { theta.sin() }

pub trait Angle<T>: Trig<T> {}
