fn main() {
  auto s = #shell { uname -a && hg identify };
  log s;
}
