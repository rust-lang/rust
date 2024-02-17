//@ check-pass

fn main() {}

fn filter_fold<T, Acc, PRED: FnMut(&T) -> bool, FOLD: FnMut(Acc, T) -> Acc>(
    mut predicate: PRED,
    mut fold: FOLD,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, item| if predicate(&item) { fold(acc, item) } else { acc }
}
