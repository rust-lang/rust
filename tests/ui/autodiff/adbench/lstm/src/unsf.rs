//@ ignore-auxiliary lib.rs
use std::autodiff::autodiff_reverse;

// Sigmoid on scalar
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// log(sum(exp(x), 2))
unsafe fn logsumexp(vect: *const f64, sz: usize) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..sz {
        sum += (*vect.add(i)).exp();
    }
    sum += 2.0; // Adding 2 to sum
    sum.ln()
}

// LSTM OBJECTIVE
// The LSTM model
unsafe fn lstm_model(
    hsize: usize,
    weight: *const f64,
    bias: *const f64,
    hidden: *mut f64,
    cell: *mut f64,
    input: *const f64,
) {
    //    // NOTE THIS
    //    //__builtin_assume(hsize > 0);
    let mut gates = vec![0.0; 4 * hsize];
    let forget: *mut f64 = gates.as_mut_ptr();
    let ingate: *mut f64 = gates[hsize..].as_mut_ptr();
    let outgate: *mut f64 = gates[2 * hsize..].as_mut_ptr();
    let change: *mut f64 = gates[3 * hsize..].as_mut_ptr();
    //let (a,b) = gates.split_at_mut(2*hsize);
    //let ((forget, ingate), (outgate, change)) = (
    //    a.split_at_mut(hsize), b.split_at_mut(hsize));

    // caching input
    for i in 0..hsize {
        *forget.add(i) = sigmoid(*input.add(i) * *weight.add(i) + *bias.add(i));
        *ingate.add(i) = sigmoid(*hidden.add(i) * *weight.add(hsize + i) + *bias.add(hsize + i));
        *outgate.add(i) =
            sigmoid(*input.add(i) * *weight.add(2 * hsize + i) + *bias.add(2 * hsize + i));
        *change.add(i) =
            (*hidden.add(i) * *weight.add(3 * hsize + i) + *bias.add(3 * hsize + i)).tanh();
    }

    // caching cell
    for i in 0..hsize {
        *cell.add(i) = *cell.add(i) * *forget.add(i) + *ingate.add(i) * *change.add(i);
    }

    for i in 0..hsize {
        *hidden.add(i) = *outgate.add(i) * (*cell.add(i)).tanh();
    }
}

// Predict LSTM output given an input
unsafe fn lstm_predict(
    l: usize,
    b: usize,
    w: *const f64,
    w2: *const f64,
    s: *mut f64,
    x: *const f64,
    x2: *mut f64,
) {
    for i in 0..b {
        *x2.add(i) = *x.add(i) * *w2.add(i);
    }

    let mut xp = x2;
    let stop = 2 * l * b;
    for i in (0..=stop - 1).step_by(2 * b) {
        lstm_model(
            b,
            w.add(i * 4),
            w.add((i + b) * 4),
            s.add(i),
            s.add(i + b),
            xp,
        );
        xp = s.add(i);
    }

    for i in 0..b {
        *x2.add(i) = *xp.add(i) * *w2.add(b + i) + *w2.add(2 * b + i);
    }
}

// LSTM objective (loss function)
#[autodiff_reverse(
    d_lstm_unsafe_objective,
    Const,
    Const,
    Const,
    Duplicated,
    Duplicated,
    Const,
    Const,
    DuplicatedOnly
)]
pub(crate) unsafe fn lstm_unsafe_objective(
    l: usize,
    c: usize,
    b: usize,
    main_params: *const f64,
    extra_params: *const f64,
    state: *mut f64,
    sequence: *const f64,
    loss: *mut f64,
) {
    let mut total = 0.0;
    let mut count = 0;

    //const double* input = &(sequence[0]);
    let mut input = sequence;
    let mut ypred = vec![0.0; b];
    let mut ynorm = vec![0.0; b];
    let mut lse;

    assert!(b > 0);

    let stop = (c - 1) * b;
    for t in (0..=stop - 1).step_by(b) {
        lstm_predict(
            l,
            b,
            main_params,
            extra_params,
            state,
            input,
            ypred.as_mut_ptr(),
        );
        lse = logsumexp(ypred.as_mut_ptr(), b);
        for i in 0..b {
            ynorm[i] = ypred[i] - lse;
        }

        //let ygold = &sequence[t + b..];
        let ygold = sequence.add(t + b);
        for i in 0..b {
            total += *ygold.add(i) * ynorm[i];
        }

        count += b;
        input = ygold;
    }

    *loss = -total / count as f64;
}
