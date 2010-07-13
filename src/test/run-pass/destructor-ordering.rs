// We share an instance of this type among all the destructor-order
// checkers.  It tracks how many destructors have run so far and
// 'fail's when one runs out of order.
// FIXME: Make it easier to collect a failure message.
state obj order_tracker(mutable int init) {
  fn assert_order(int expected, str fail_message) {
    if (expected != init) {
      log expected;
      log " != ";
      log init;
      log fail_message;
      fail;
    }
    init += 1;
  }
}


obj dorder(@order_tracker tracker, int order, str message) {
  drop {
    tracker.assert_order(order, message);
  }
}

fn main() {
  auto tracker = @order_tracker(0);
  dorder(tracker, 1, "Reverse decl order");
  dorder(tracker, 0, "Reverse decl order");
}