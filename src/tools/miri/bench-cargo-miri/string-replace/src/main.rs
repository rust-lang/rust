const TCB_INFO_JSON: &str = include_str!("../data.json");

fn main() {
    let tcb_json = TCB_INFO_JSON;
    let bad_tcb_json = tcb_json.replace("female", "male");
    std::hint::black_box(bad_tcb_json);
}
