// pretty-expanded FIXME #23616

struct cat {

  name : String,

}

fn cat(in_name: String) -> cat {
    cat {
        name: in_name
    }
}

pub fn main() {
  let _nyan = cat("nyan".to_string());
}
