fn main() {
  let isize x = 5; //~ ERROR expected one of `:`, `;`, `=`, or `@`, found `x`
  match x;
}
