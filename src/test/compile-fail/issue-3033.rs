type cat = {cat_name: ~str, cat_name: int};  //~ ERROR Duplicate field name cat_name

fn main()
{
  io::println(int::str({x: 1, x: 2}.x)); //~ ERROR Duplicate field name x
}
