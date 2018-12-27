// Related issues: #20401, #20506, #20614, #20752, #20829, #20846, #20885, #20886

fn main() {
    "".homura[""]; //~ no field `homura` on type `&'static str`
}
