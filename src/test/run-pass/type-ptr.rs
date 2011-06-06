fn f(*int a) -> *int {
   ret a;
}

fn g(*int a) -> *int {
   auto b = f(a);
   ret b;
}

fn main(vec[str] args) {
  ret;
}
