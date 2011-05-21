// When all branches of an alt expression result in fail, the entire
// alt expression results in fail.

fn main() {
  auto x = alt (true) {
    case (true) {
      10
    }
    case (true) {
      alt (true) {
        case (true) {
          fail
        }
        case (false) {
          fail
        }
      }
    }
  };
}
