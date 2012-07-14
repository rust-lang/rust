class cat {

  let name : ~str;

  new(in_name: ~str)
    { self.name = in_name; }
}

fn main() {
  let nyan = cat(~"nyan");
}