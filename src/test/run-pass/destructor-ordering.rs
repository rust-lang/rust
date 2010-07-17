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
    (*tracker).assert_order(order, message);
  }
}

fn test_simple() {
  auto tracker = @order_tracker(0);
  dorder(tracker, 1, "Reverse decl order");
  dorder(tracker, 0, "Reverse decl order");
}

fn test_block() {
  auto tracker = @order_tracker(0);
  dorder(tracker, 2, "Before block");
  {
    dorder(tracker, 0, "Inside block");
  }
  dorder(tracker, 1, "After block");
}

fn test_decl_v_init() {
  auto tracker = @order_tracker(0);
  auto var1;
  auto var2;
  var2 = dorder(tracker, 0, "decl, not init");
  var1 = dorder(tracker, 1, "decl, not init");
}

fn test_overwritten_obj() {
  auto tracker = @order_tracker(0);
  auto var1 = dorder(tracker, 0, "overwritten object destroyed first");
  auto var2 = dorder(tracker, 2, "destroyed at end of scope");
  var1 = dorder(tracker, 3, "overwriter deleted in rev decl order");
  {
    dorder(tracker, 1, "overwritten object destroyed before end of scope");
  }
}

// Used to embed dorder objects into an expression.  Note that the
// parameters don't get destroyed.
fn combine_dorders(dorder d1, dorder d2) -> int {
  ret 1;
}
fn test_expression_destroyed_right_to_left() {
  auto tracker = @order_tracker(0);
  combine_dorders(dorder(tracker, 4, ""), dorder(tracker, 3, ""))
    / combine_dorders(dorder(tracker, 2, ""), dorder(tracker, 1, ""));
  {
    dorder(tracker, 0,
           "expression objects live to end of block, not statement");
  }
}

fn main() {
  test_simple();
  test_block();
  test_decl_v_init();
  test_overwritten_obj();
  test_expression_destroyed_right_to_left();
}
