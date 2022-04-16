use std::cmp;

use rustc_middle::ty::error::TypeError;

// An issue that might be found in the compatibility matrix
enum Issue {
    /// The given argument is the invalid type for the input
    Invalid(usize),
    /// There is a missing input
    Missing(usize),
    /// There's a superfluous argument
    Extra(usize),
    /// Two arguments should be swapped
    Swap(usize, usize),
    /// Several arguments should be reordered
    Permutation(Vec<Option<usize>>),
}

#[derive(Clone, Debug)]
pub(crate) enum Compatibility<'tcx> {
    Compatible,
    Incompatible(Option<TypeError<'tcx>>),
}

/// Similar to `Issue`, but contains some extra information
pub(crate) enum Error<'tcx> {
    /// The given argument is the invalid type for the input
    Invalid(usize, Compatibility<'tcx>),
    /// There is a missing input
    Missing(usize),
    /// There's a superfluous argument
    Extra(usize),
    /// Two arguments should be swapped
    Swap(usize, usize, usize, usize),
    /// Several arguments should be reordered
    Permutation(Vec<(usize, usize)>), // dest_arg, dest_input
}

pub(crate) struct ArgMatrix<'tcx> {
    input_indexes: Vec<usize>,
    arg_indexes: Vec<usize>,
    compatibility_matrix: Vec<Vec<Compatibility<'tcx>>>,
}

impl<'tcx> ArgMatrix<'tcx> {
    pub(crate) fn new<F: FnMut(usize, usize) -> Compatibility<'tcx>>(
        minimum_input_count: usize,
        provided_arg_count: usize,
        mut is_compatible: F,
    ) -> Self {
        let compatibility_matrix = (0..provided_arg_count)
            .map(|i| (0..minimum_input_count).map(|j| is_compatible(i, j)).collect())
            .collect();
        ArgMatrix {
            input_indexes: (0..minimum_input_count).collect(),
            arg_indexes: (0..provided_arg_count).collect(),
            compatibility_matrix,
        }
    }

    /// Remove a given input from consideration
    fn eliminate_input(&mut self, idx: usize) {
        self.input_indexes.remove(idx);
        for row in &mut self.compatibility_matrix {
            row.remove(idx);
        }
    }

    /// Remove a given argument from consideration
    fn eliminate_arg(&mut self, idx: usize) {
        self.arg_indexes.remove(idx);
        self.compatibility_matrix.remove(idx);
    }

    /// "satisfy" an input with a given arg, removing both from consideration
    fn satisfy_input(&mut self, input_idx: usize, arg_idx: usize) {
        self.eliminate_input(input_idx);
        self.eliminate_arg(arg_idx);
    }

    fn eliminate_satisfied(&mut self) -> Vec<(usize, usize)> {
        let mut i = cmp::min(self.input_indexes.len(), self.arg_indexes.len());
        let mut eliminated = vec![];
        while i > 0 {
            let idx = i - 1;
            if matches!(self.compatibility_matrix[idx][idx], Compatibility::Compatible) {
                eliminated.push((self.arg_indexes[idx], self.input_indexes[idx]));
                self.satisfy_input(idx, idx);
            }
            i -= 1;
        }
        return eliminated;
    }

    // Check for the above mismatch cases
    fn find_issue(&self) -> Option<Issue> {
        let mat = &self.compatibility_matrix;
        let ai = &self.arg_indexes;
        let ii = &self.input_indexes;

        for i in 0..cmp::max(ai.len(), ii.len()) {
            // If we eliminate the last row, any left-over inputs are considered missing
            if i >= mat.len() {
                return Some(Issue::Missing(i));
            }
            // If we eliminate the last column, any left-over arguments are extra
            if mat[i].len() == 0 {
                return Some(Issue::Extra(i));
            }

            // Make sure we don't pass the bounds of our matrix
            let is_arg = i < ai.len();
            let is_input = i < ii.len();
            if is_arg && is_input && matches!(mat[i][i], Compatibility::Compatible) {
                // This is a satisfied input, so move along
                continue;
            }

            let mut useless = true;
            let mut unsatisfiable = true;
            if is_arg {
                for j in 0..ii.len() {
                    // If we find at least one input this argument could satisfy
                    // this argument isn't completely useless
                    if matches!(mat[i][j], Compatibility::Compatible) {
                        useless = false;
                        break;
                    }
                }
            }
            if is_input {
                for j in 0..ai.len() {
                    // If we find at least one argument that could satisfy this input
                    // this argument isn't unsatisfiable
                    if matches!(mat[j][i], Compatibility::Compatible) {
                        unsatisfiable = false;
                        break;
                    }
                }
            }

            match (is_arg, is_input, useless, unsatisfiable) {
                // If an input is unsatisfied, and the argument in its position is useless
                // then the most likely explanation is that we just got the types wrong
                (true, true, true, true) => return Some(Issue::Invalid(i)),
                // Otherwise, if an input is useless, then indicate that this is an extra argument
                (true, _, true, _) => return Some(Issue::Extra(i)),
                // Otherwise, if an argument is unsatisfiable, indicate that it's missing
                (_, true, _, true) => return Some(Issue::Missing(i)),
                (true, true, _, _) => {
                    // The argument isn't useless, and the input isn't unsatisfied,
                    // so look for a parameter we might swap it with
                    // We look for swaps explicitly, instead of just falling back on permutations
                    // so that cases like (A,B,C,D) given (B,A,D,C) show up as two swaps,
                    // instead of a large permutation of 4 elements.
                    for j in 0..cmp::min(ai.len(), ii.len()) {
                        if i == j || matches!(mat[j][j], Compatibility::Compatible) {
                            continue;
                        }
                        if matches!(mat[i][j], Compatibility::Compatible)
                            && matches!(mat[j][i], Compatibility::Compatible)
                        {
                            return Some(Issue::Swap(i, j));
                        }
                    }
                }
                _ => {
                    continue;
                }
            };
        }

        // We didn't find any of the individual issues above, but
        // there might be a larger permutation of parameters, so we now check for that
        // by checking for cycles
        // We use a double option at position i in this vec to represent:
        // - None: We haven't computed anything about this argument yet
        // - Some(None): This argument definitely doesn't participate in a cycle
        // - Some(Some(x)): the i-th argument could permute to the x-th position
        let mut permutation: Vec<Option<Option<usize>>> = vec![None; mat.len()];
        let mut permutation_found = false;
        for i in 0..mat.len() {
            if permutation[i].is_some() {
                // We've already decided whether this argument is or is not in a loop
                continue;
            }

            let mut stack = vec![];
            let mut j = i;
            let mut last = i;
            let mut is_cycle = true;
            loop {
                stack.push(j);
                // Look for params this one could slot into
                let compat: Vec<_> =
                    mat[j]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| {
                            if matches!(c, Compatibility::Compatible) { Some(i) } else { None }
                        })
                        .collect();
                if compat.len() != 1 {
                    // this could go into multiple slots, don't bother exploring both
                    is_cycle = false;
                    break;
                }
                j = compat[0];
                if stack.contains(&j) {
                    last = j;
                    break;
                }
            }
            if stack.len() <= 2 {
                // If we encounter a cycle of 1 or 2 elements, we'll let the
                // "satisfy" and "swap" code above handle those
                is_cycle = false;
            }
            // We've built up some chain, some of which might be a cycle
            // ex: [1,2,3,4]; last = 2; j = 2;
            // So, we want to mark 4, 3, and 2 as part of a permutation
            permutation_found = is_cycle;
            while let Some(x) = stack.pop() {
                if is_cycle {
                    permutation[x] = Some(Some(j));
                    j = x;
                    if j == last {
                        // From here on out, we're a tail leading into a cycle,
                        // not the cycle itself
                        is_cycle = false;
                    }
                } else {
                    // Some(None) ensures we save time by skipping this argument again
                    permutation[x] = Some(None);
                }
            }
        }

        if permutation_found {
            // Map unwrap to remove the first layer of Some
            let final_permutation: Vec<Option<usize>> =
                permutation.into_iter().map(|x| x.unwrap()).collect();
            return Some(Issue::Permutation(final_permutation));
        }
        return None;
    }

    // Obviously, detecting exact user intention is impossible, so the goal here is to
    // come up with as likely of a story as we can to be helpful.
    //
    // We'll iteratively removed "satisfied" input/argument pairs,
    // then check for the cases above, until we've eliminated the entire grid
    //
    // We'll want to know which arguments and inputs these rows and columns correspond to
    // even after we delete them.
    pub(crate) fn find_errors(mut self) -> (Vec<Error<'tcx>>, Vec<Option<usize>>) {
        let provided_arg_count = self.arg_indexes.len();

        let mut errors: Vec<Error<'tcx>> = vec![];
        // For each expected argument, the matched *actual* input
        let mut matched_inputs: Vec<Option<usize>> = vec![None; self.input_indexes.len()];

        // Before we start looking for issues, eliminate any arguments that are already satisfied,
        // so that an argument which is already spoken for by the input it's in doesn't
        // spill over into another similarly typed input
        // ex:
        //   fn some_func(_a: i32, _b: i32) {}
        //   some_func(1, "");
        // Without this elimination, the first argument causes the second argument
        // to show up as both a missing input and extra argument, rather than
        // just an invalid type.
        for (arg, inp) in self.eliminate_satisfied() {
            matched_inputs[inp] = Some(arg);
        }

        while self.input_indexes.len() > 0 || self.arg_indexes.len() > 0 {
            // Check for the first relevant issue
            match self.find_issue() {
                Some(Issue::Invalid(idx)) => {
                    let compatibility = self.compatibility_matrix[idx][idx].clone();
                    let input_idx = self.input_indexes[idx];
                    self.satisfy_input(idx, idx);
                    errors.push(Error::Invalid(input_idx, compatibility));
                }
                Some(Issue::Extra(idx)) => {
                    let arg_idx = self.arg_indexes[idx];
                    self.eliminate_arg(idx);
                    errors.push(Error::Extra(arg_idx));
                }
                Some(Issue::Missing(idx)) => {
                    let input_idx = self.input_indexes[idx];
                    self.eliminate_input(idx);
                    errors.push(Error::Missing(input_idx));
                }
                Some(Issue::Swap(idx, other)) => {
                    let input_idx = self.input_indexes[idx];
                    let other_input_idx = self.input_indexes[other];
                    let arg_idx = self.arg_indexes[idx];
                    let other_arg_idx = self.arg_indexes[other];
                    let (min, max) = (cmp::min(idx, other), cmp::max(idx, other));
                    self.satisfy_input(min, max);
                    // Subtract 1 because we already removed the "min" row
                    self.satisfy_input(max - 1, min);
                    errors.push(Error::Swap(input_idx, other_input_idx, arg_idx, other_arg_idx));
                    matched_inputs[input_idx] = Some(other_arg_idx);
                    matched_inputs[other_input_idx] = Some(arg_idx);
                }
                Some(Issue::Permutation(args)) => {
                    // FIXME: If satisfy_input ever did anything non-trivial (emit obligations to help type checking, for example)
                    // we'd want to call this function with the correct arg/input pairs, but for now, we just throw them in a bucket.
                    // This works because they force a cycle, so each row is guaranteed to also be a column
                    let mut idxs: Vec<usize> = args.iter().filter_map(|&a| a).collect();

                    let mut real_idxs = vec![None; provided_arg_count];
                    for (src, dst) in
                        args.iter().enumerate().filter_map(|(src, dst)| dst.map(|dst| (src, dst)))
                    {
                        let src_arg = self.arg_indexes[src];
                        let dst_arg = self.arg_indexes[dst];
                        let dest_input = self.input_indexes[dst];
                        real_idxs[src_arg] = Some((dst_arg, dest_input));
                        matched_inputs[dest_input] = Some(src_arg);
                    }
                    idxs.sort();
                    idxs.reverse();
                    for i in idxs {
                        self.satisfy_input(i, i);
                    }
                    errors.push(Error::Permutation(real_idxs.into_iter().flatten().collect()));
                }
                None => {
                    // We didn't find any issues, so we need to push the algorithm forward
                    // First, eliminate any arguments that currently satisfy their inputs
                    for (arg, inp) in self.eliminate_satisfied() {
                        matched_inputs[inp] = Some(arg);
                    }
                }
            };
        }

        return (errors, matched_inputs);
    }
}
