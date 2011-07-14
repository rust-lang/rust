// xfail-stage0

fn main() {
  #macro([#mylambda(x,body), {fn f(int x) -> int { ret body }; f}]);

  assert(#mylambda(y,y*2)(8) == 16);
}