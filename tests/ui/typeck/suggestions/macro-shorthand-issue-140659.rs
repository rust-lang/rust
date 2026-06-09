trait Reencode {
    type Error;
    fn tag_index(&mut self, tag: u32) -> Result<u32, Self::Error>;
}

struct Reencoder;
impl Reencode for Reencoder {
    type Error = &'static str;
    fn tag_index(&mut self, tag: u32) -> Result<u32, Self::Error> {
        Ok(tag)
    }
}


enum Operator {
    Suspend { tag_index: u32 },
}

enum Instruction {
    Suspend { tag_index: u32 },
}


macro_rules! for_each_operator {
    ($m:ident) => {
        $m! {
            Suspend { tag_index: u32 } => visit_suspend
        }
    };
}


fn process<T: Reencode>(op: &Operator, reencoder: &mut T) -> Instruction {
    macro_rules! translate {
        (Suspend { tag_index: $ty:ty } => $visit:ident) => {
            match op {
                Operator::Suspend { tag_index } => {
                    let tag_index = reencoder.tag_index(*tag_index);

                    // KEY POINT: Using field shorthand syntax where the compiler gets confused
                    // Here tag_index is a Result<u32, E> but we're using it where u32 is expected
                    Instruction::Suspend { tag_index } //~ ERROR mismatched types [E0308]
                }
            }
        };
    }

    for_each_operator!(translate)
}

fn main() {
    let mut reencoder = Reencoder;
    let op = Operator::Suspend { tag_index: 1 };

    let _ = process(&op, &mut reencoder);
}
