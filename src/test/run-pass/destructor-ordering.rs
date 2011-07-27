// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// This test checks that destructors run in the right order.  Because
// stateful objects can't have destructors, we have the destructors
// record their expected order into a channel when they execute (so
// the object becomes 'io' rather than 'state').  Then each test case
// asserts that the channel produces values in ascending order.
//
// FIXME: Write an int->str function and concatenate the whole failure
// message into a single log statement (or, even better, a print).
//
// FIXME: check_order should take only 1 line in a test, not 2+a block
// block. Since destructor-having objects can't refer to mutable state
// (like the port), we'd need a with-like construct to do the same for
// stateful objects within a scope.
//
// FIXME #21: Each test should execute in its own task, so it can fail
// independently, writing its error message to a channel that the
// parent task aggregates.

type order_info = rec(int order, str msg);

io fn check_order(port[order_info] expected_p) {
  chan(expected_p) <| rec(order=-1, msg="");
  let mutable int actual = 0;
  // FIXME #121: Workaround for while(true) bug.
  auto expected; expected_p |> expected;
  auto done = -1;  // FIXME: Workaround for typechecking bug.
  while(expected.order != done) {
    if (expected.order != actual) {
      log expected.order;
      log " != ";
      log actual;
      log expected.msg;
      fail;
    }
    actual += 1;
    expected_p |> expected;
  }
}


obj dorder(chan[order_info] expected, int order, str message) {
  drop {
    expected <| rec(order=order, msg=message);
  }
}

io fn test_simple() {
  let port[order_info] tracker_p = port();
  auto tracker = chan(tracker_p);
  dorder(tracker, 1, "Reverse decl order");
  dorder(tracker, 0, "Reverse decl order");
  check_order(tracker_p);
}

io fn test_block() {
  let port[order_info] tracker_p = port();
  auto tracker = chan(tracker_p);
  {
    dorder(tracker, 2, "Before block");
    {
      dorder(tracker, 0, "Inside block");
    }
    dorder(tracker, 1, "After block");
  }
  check_order(tracker_p);
}

io fn test_decl_v_init() {
  let port[order_info] tracker_p = port();
  auto tracker = chan(tracker_p);
  {
    auto var1;
    auto var2;
    var2 = dorder(tracker, 0, "decl, not init");
    var1 = dorder(tracker, 1, "decl, not init");
  }
  check_order(tracker_p);
}

io fn test_overwritten_obj() {
  let port[order_info] tracker_p = port();
  auto tracker = chan(tracker_p);
  {
    auto var1 = dorder(tracker, 0, "overwritten object destroyed first");
    auto var2 = dorder(tracker, 2, "destroyed at end of scope");
    var1 = dorder(tracker, 3, "overwriter deleted in rev decl order");
    {
      dorder(tracker, 1, "overwritten object destroyed before end of scope");
    }
  }
  check_order(tracker_p);
}

// Used to embed dorder objects into an expression.  Note that the
// parameters don't get destroyed.
fn combine_dorders(dorder d1, dorder d2) -> int {
  ret 1;
}
io fn test_expression_destroyed_right_to_left() {
  let port[order_info] tracker_p = port();
  auto tracker = chan(tracker_p);
  {
    combine_dorders(dorder(tracker, 4, ""), dorder(tracker, 3, ""))
      / combine_dorders(dorder(tracker, 2, ""), dorder(tracker, 1, ""));
    {
      dorder(tracker, 0,
             "expression objects live to end of block, not statement");
    }
  }
  check_order(tracker_p);
}

io fn main() {
  test_simple();
  test_block();
  test_decl_v_init();
  test_overwritten_obj();
  test_expression_destroyed_right_to_left();
}
