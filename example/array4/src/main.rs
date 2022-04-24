struct soultion;
impl soultion{
    pub fn move_zero(nums:&mut Vec<i32>){
        let mut i=0;
        for j in 0..nums.len(){
            if nums[j] !=0{
                nums[i] = nums[j];
                i+=1;
            }
        }
   for k in i ..nums.len(){
       nums[k] = 0;
   }
    }
}
fn main() {
    let mut v = vec![1,23,0,0,0,3,4,5];
    soultion::move_zero( &mut v);
    println!("v:{:?}",v);
}
