// build-pass (FIXME(62277): could be check-pass?)
// https://github.com/rust-lang/rust/issues/51300

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Stat {
    pub id: u8,
    pub index: usize,
}

impl Stat {
    pub const STUDENT_HAPPINESS: Stat = Stat{
        id: 0,
        index: 0,
    };
    pub const STUDENT_HUNGER: Stat = Stat{
        id: 0,
        index: Self::STUDENT_HAPPINESS.index + 1,
    };

}

pub fn from_index(id: u8, index: usize) -> Option<Stat> {
    let stat = Stat{id, index};
    match stat {
        Stat::STUDENT_HAPPINESS => Some(Stat::STUDENT_HAPPINESS),
        Stat::STUDENT_HUNGER => Some(Stat::STUDENT_HUNGER),
        _ => None,
    }
}

fn main() { }
