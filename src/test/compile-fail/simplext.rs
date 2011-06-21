//error-pattern:expects 0 arguments, got 16

fn main() {
  #simplext("trivial", 1*2*4*2*1);
  
  assert(#trivial(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) == 16);
}
