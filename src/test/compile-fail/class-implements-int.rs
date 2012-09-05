struct cat : int { //~ ERROR trait
  let meows: uint;
}

fn cat(in_x : uint) -> cat {
    cat {
        meows: in_x
    }
}

fn main() {
  let nyan = cat(0u);
}
