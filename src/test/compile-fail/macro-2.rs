//error-pattern:expanded as an identifier
fn main() {
  #macro([#mylambda(x, body), {fn f(int x) -> int {ret body}; f}]);

  assert(#mylambda(y*1, y*2)(8) == 16);
}