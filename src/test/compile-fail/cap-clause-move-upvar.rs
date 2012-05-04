fn main() {
    let x = 5;
    let _y = fn~(move x) -> int {
        let _z = fn~(move x) -> int { x }; //! ERROR moving out of upvar
        22
    };
}
