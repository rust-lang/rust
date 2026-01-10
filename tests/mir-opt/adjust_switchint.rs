//@ test-mir-pass: AdjustDiscriminantSwitches

pub enum Niched1 {
    A,
    B(bool),
    C,
}

// EMIT_MIR adjust_switchint.niched_1.AdjustDiscriminantSwitches.diff
pub fn niched_1(e: Niched1) -> u32 {
    // CHECK: [[DISCR:_.+]] = discriminant(_1);
    // CHECK: [[ADJ:_.+]] = Add(move [[DISCR]], const 2_isize);
    // CHECK: switchInt(move [[ADJ]]) -> [2: {{bb.+}}, 3: {{bb.+}}, 4: {{bb.+}}, otherwise: {{bb.+}}];
    match e {
        Niched1::A => 7,
        Niched1::B(b) => b as _,
        Niched1::C => 42,
    }
}

pub fn main() {
    niched_1(Niched1::A);
}
