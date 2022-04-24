struct soultion;

impl soultion{
    pub fn zero_move(nums: & mut Vec<i32>){
        let mut i =0;
        let mut j =0;
        while j <nums.len(){
            if nums[j] !=0{
                nums[i] = nums[j];
                i+=1;
            }
            j +=1;
        }
        let mut k = i;
        while k < nums.len() {
            nums[k] =0;
            k +=1;
            
        }

    }
}
fn main() {
    let mut vec:Vec<i32> = vec![0,0,3,0,5];
    soultion::zero_move(& mut vec);
    println!("{:?}",vec);
}
