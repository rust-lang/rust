//@ run-pass
// Test equality constrai32s on associated types in a where clause.


pub trait ToI32 {
    fn to_i32(&self) -> i32;
}

impl ToI32 for i32 {
    fn to_i32(&self) -> i32 { *self }
}

impl ToI32 for u32 {
    fn to_i32(&self) -> i32 { *self as i32 }
}

pub trait GetToI32
{
    type R : ToI32;

    fn get(&self) -> <Self as GetToI32>::R;
}

impl GetToI32 for i32 {
    type R = i32;
    fn get(&self) -> i32 { *self }
}

impl GetToI32 for u32 {
    type R = u32;
    fn get(&self) -> u32 { *self }
}

fn foo<G>(g: G) -> i32
    where G : GetToI32
{
    ToI32::to_i32(&g.get())
}

pub fn main() {
    assert_eq!(foo(22i32), 22);
    assert_eq!(foo(22u32), 22);
}
