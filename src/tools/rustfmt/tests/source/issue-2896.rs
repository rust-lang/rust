extern crate rand;
extern crate timely;
extern crate differential_dataflow;

use rand::{Rng, SeedableRng, StdRng};

use timely::dataflow::operators::*;

use differential_dataflow::AsCollection;
use differential_dataflow::operators::*;
use differential_dataflow::input::InputSession;

// mod loglikelihoodratio;

fn main() {

  // define a new timely dataflow computation. 
  timely::execute_from_args(std::env::args().skip(6), move |worker| {

    // capture parameters of the experiment.
    let users: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let items: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let scale: usize = std::env::args().nth(3).unwrap().parse().unwrap();
    let batch: usize = std::env::args().nth(4).unwrap().parse().unwrap();
    let noisy: bool = std::env::args().nth(5).unwrap() == "noisy";

    let index = worker.index();
    let peers = worker.peers();

    let (input, probe) = worker.dataflow(|scope| {

      // input of (user, item) collection.
      let (input, occurrences) = scope.new_input();
      let occurrences = occurrences.as_collection();

      //TODO adjust code to only work with upper triangular half of cooccurrence matrix

      /* Compute the cooccurrence matrix C = A'A from the binary interaction matrix A. */
      let cooccurrences = 
      occurrences
        .join_map(&occurrences, |_user, &item_a, &item_b| (item_a, item_b))
        .filter(|&(item_a, item_b)| item_a != item_b)
        .count();

      /* compute the rowsums of C indicating how often we encounter individual items. */
      let row_sums = 
      occurrences
        .map(|(_user, item)| item)
        .count();

      // row_sums.inspect(|record| println!("[row_sums] {:?}", record));

      /* Join the cooccurrence pairs with the corresponding row sums. */
      let mut cooccurrences_with_row_sums = cooccurrences
        .map(|((item_a, item_b), num_cooccurrences)| (item_a, (item_b, num_cooccurrences)))
        .join_map(&row_sums, |&item_a, &(item_b, num_cooccurrences), &row_sum_a| {
          assert!(row_sum_a > 0);
          (item_b, (item_a, num_cooccurrences, row_sum_a))
        })
        .join_map(&row_sums, |&item_b, &(item_a, num_cooccurrences, row_sum_a), &row_sum_b| {
          assert!(row_sum_a > 0);
          assert!(row_sum_b > 0);
          (item_a, (item_b, num_cooccurrences, row_sum_a, row_sum_b))
        });

      // cooccurrences_with_row_sums
      //     .inspect(|record| println!("[cooccurrences_with_row_sums] {:?}", record));

      // //TODO compute top-k "similar items" per item
      // /* Compute LLR scores for each item pair. */
      // let llr_scores = cooccurrences_with_row_sums.map(
      //   |(item_a, (item_b, num_cooccurrences, row_sum_a, row_sum_b))| {

      //     println!(
      //       "[llr_scores] item_a={} item_b={}, num_cooccurrences={} row_sum_a={} row_sum_b={}",
      //       item_a, item_b, num_cooccurrences, row_sum_a, row_sum_b);

      //     let k11: isize = num_cooccurrences;
      //     let k12: isize = row_sum_a as isize - k11;
      //     let k21: isize = row_sum_b as isize - k11;
      //     let k22: isize = 10000 - k12 - k21 + k11;

      //     let llr_score = loglikelihoodratio::log_likelihood_ratio(k11, k12, k21, k22);

      //     ((item_a, item_b), llr_score)
      //   });

      if noisy {
        cooccurrences_with_row_sums = 
        cooccurrences_with_row_sums
          .inspect(|x| println!("change: {:?}", x));
      }

      let probe = 
      cooccurrences_with_row_sums
          .probe();
/*
      // produce the (item, item) collection
      let cooccurrences = occurrences
        .join_map(&occurrences, |_user, &item_a, &item_b| (item_a, item_b));
      // count the occurrences of each item.
      let counts = cooccurrences
        .map(|(item_a,_)| item_a)
        .count();
      // produce ((item1, item2), count1, count2, count12) tuples
      let cooccurrences_with_counts = cooccurrences
        .join_map(&counts, |&item_a, &item_b, &count_item_a| (item_b, (item_a, count_item_a)))
        .join_map(&counts, |&item_b, &(item_a, count_item_a), &count_item_b| {
          ((item_a, item_b), count_item_a, count_item_b)
        });
      let probe = cooccurrences_with_counts
        .inspect(|x| println!("change: {:?}", x))
        .probe();
*/
      (input, probe)
    });

    let seed: &[_] = &[1, 2, 3, index];
    let mut rng1: StdRng = SeedableRng::from_seed(seed);  // rng for edge additions
    let mut rng2: StdRng = SeedableRng::from_seed(seed);  // rng for edge deletions

    let mut input = InputSession::from(input);

    for count in 0 .. scale {
      if count % peers == index {
        let user = rng1.gen_range(0, users);
        let item = rng1.gen_range(0, items);
        // println!("[INITIAL INPUT] ({}, {})", user, item);
        input.insert((user, item));
      }
    }

    // load the initial data up!
    while probe.less_than(input.time()) { worker.step(); }

    for round in 1 .. {

      for element in (round * batch) .. ((round + 1) * batch) {
        if element % peers == index {
          // advance the input timestamp.
          input.advance_to(round * batch);
          // insert a new item.
          let user = rng1.gen_range(0, users);
          let item = rng1.gen_range(0, items);
          if noisy { println!("[INPUT: insert] ({}, {})", user, item); }
          input.insert((user, item));
          // remove an old item.
          let user = rng2.gen_range(0, users);
          let item = rng2.gen_range(0, items);
          if noisy { println!("[INPUT: remove] ({}, {})", user, item); }
          input.remove((user, item));
        }
      }

      input.advance_to(round * batch);
      input.flush();

      while probe.less_than(input.time()) { worker.step(); }
    }
  }).unwrap();
}
