#![feature(unsized_fn_params)]

pub fn udrop<T: ?Sized>(_x: T) {}
