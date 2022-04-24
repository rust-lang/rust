use std :: ops::Add;
// 范型约束
fn sum<T: Add<T,Output = T>>(a:T,b:T)->T {
    a +b
    
}
fn main() {
    assert_eq!(sum(1u32,2u32),3);
    assert_eq!(sum(1u64,2u64),3);
}
