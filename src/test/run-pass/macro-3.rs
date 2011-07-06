fn main() {
  #macro([#trivial(), 1*2*4*2*1]);
  
  assert(#trivial() == 16);
}
