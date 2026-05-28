// Test /* comment */ inside trait generics does not get duplicated.
trait Test</* comment */ T> {}

trait TestTwo</* comment */ T, /* comment */ V> {}
