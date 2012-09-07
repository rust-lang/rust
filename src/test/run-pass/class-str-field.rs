struct cat {

  name : ~str,

}

fn cat(in_name: ~str) -> cat {
    cat {
        name: in_name
    }
}

fn main() {
  let nyan = cat(~"nyan");
}