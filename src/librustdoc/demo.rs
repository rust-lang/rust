// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-reformat

/*!
 * A demonstration module
 *
 * Contains documentation in various forms that rustdoc understands,
 * for testing purposes. It doesn't surve any functional
 * purpose. This here, for instance, is just some filler text.
 *
 * FIXME (#3731): It would be nice if we could run some automated
 * tests on this file
 */

use core::prelude::*;

/// The base price of a muffin on a non-holiday
static price_of_a_muffin: float = 70f;

struct WaitPerson {
    hair_color: ~str
}

/// The type of things that produce omnomnom
enum OmNomNomy {
    /// Delicious sugar cookies
    Cookie,
    /// It's pizza
    PizzaPie(~[uint])
}

fn take_my_order_please(
    _waitperson: WaitPerson,
    _order: ~[OmNomNomy]
) -> uint {

    /*!
     * OMG would you take my order already?
     *
     * # Arguments
     *
     * * _waitperson - The waitperson that you want to bother
     * * _order - The order vector. It should be filled with food
     *
     * # Return
     *
     * The price of the order, including tax
     *
     * Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec
     * molestie nisl. Duis massa risus, pharetra a scelerisque a,
     * molestie eu velit. Donec mattis ligula at ante imperdiet ut
     * dapibus mauris malesuada.
     *
     * Sed gravida nisi a metus elementum sit amet hendrerit dolor
     * bibendum. Aenean sit amet neque massa, sed tempus tortor. Sed ut
     * lobortis enim. Proin a mauris quis nunc fermentum ultrices eget a
     * erat. Mauris in lectus vitae metus sodales auctor. Morbi nunc
     * quam, ultricies at venenatis non, pellentesque ac dui.
     *
     * # Failure
     *
     * This function is full of fail
     */

    fail!();
}

mod fortress_of_solitude {
    /*!
     * Superman's vacation home
     *
     * The fortress of solitude is located in the Arctic and it is
     * cold. What you may not know about the fortress of solitude
     * though is that it contains two separate bowling alleys. One of
     * them features bumper-bowling and is kind of lame.
     *
     * Really, it's pretty cool.
     */

}

mod blade_runner {
    /*!
     * Blade Runner is probably the best movie ever
     *
     * I like that in the world of Blade Runner it is always
     * raining, and that it's always night time. And Aliens
     * was also a really good movie.
     *
     * Alien 3 was crap though.
     */
}

/**
 * Bored
 *
 * Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec
 * molestie nisl. Duis massa risus, pharetra a scelerisque a,
 * molestie eu velit. Donec mattis ligula at ante imperdiet ut
 * dapibus mauris malesuada. Sed gravida nisi a metus elementum sit
 * amet hendrerit dolor bibendum. Aenean sit amet neque massa, sed
 * tempus tortor. Sed ut lobortis enim. Proin a mauris quis nunc
 * fermentum ultrices eget a erat. Mauris in lectus vitae metus
 * sodales auctor. Morbi nunc quam, ultricies at venenatis non,
 * pellentesque ac dui.
 *
 * Quisque vitae est id eros placerat laoreet sit amet eu
 * nisi. Curabitur suscipit neque porttitor est euismod
 * lacinia. Curabitur non quam vitae ipsum adipiscing
 * condimentum. Mauris ut ante eget metus sollicitudin
 * blandit. Aliquam erat volutpat. Morbi sed nisl mauris. Nulla
 * facilisi. Phasellus at mollis ipsum. Maecenas sed convallis
 * sapien. Nullam in ligula turpis. Pellentesque a neque augue. Sed
 * eget ante feugiat tortor congue auctor ac quis ante. Proin
 * condimentum lacinia tincidunt.
 */
struct Bored {
  bored: bool,
}

impl Drop for Bored {
  fn finalize(&self) { }
}

/**
 * The Shunned House
 *
 * From even the greatest of horrors irony is seldom absent. Sometimes it
 * enters directly into the composition of the events, while sometimes it
 * relates only to their fortuitous position among persons and
 * places. The latter sort is splendidly exemplified by a case in the
 * ancient city of Providence, where in the late forties Edgar Allan Poe
 * used to sojourn often during his unsuccessful wooing of the gifted
 * poetess, Mrs.  Whitman. Poe generally stopped at the Mansion House in
 * Benefit Street--the renamed Golden Ball Inn whose roof has sheltered
 * Washington, Jefferson, and Lafayette--and his favorite walk led
 * northward along the same street to Mrs. Whitman's home and the
 * neighboring hillside churchyard of St. John's, whose hidden expanse of
 * Eighteenth Century gravestones had for him a peculiar fascination.
 */
trait TheShunnedHouse {
    /**
     * Now the irony is this. In this walk, so many times repeated, the
     * world's greatest master of the terrible and the bizarre was
     * obliged to pass a particular house on the eastern side of the
     * street; a dingy, antiquated structure perched on the abruptly
     * rising side hill, with a great unkempt yard dating from a time
     * when the region was partly open country. It does not appear that
     * he ever wrote or spoke of it, nor is there any evidence that he
     * even noticed it. And yet that house, to the two persons in
     * possession of certain information, equals or outranks in horror
     * the wildest fantasy of the genius who so often passed it
     * unknowingly, and stands starkly leering as a symbol of all that is
     * unutterably hideous.
     *
     * # Arguments
     *
     * * unkempt_yard - A yard dating from a time when the region was partly
     *                  open country
     */
    fn dingy_house(&self, unkempt_yard: int);

    /**
     * The house was--and for that matter still is--of a kind to attract
     * the attention of the curious. Originally a farm or semi-farm
     * building, it followed the average New England colonial lines of
     * the middle Eighteenth Century--the prosperous peaked-roof sort,
     * with two stories and dormerless attic, and with the Georgian
     * doorway and interior panelling dictated by the progress of taste
     * at that time. It faced south, with one gable end buried to the
     * lower windows in the eastward rising hill, and the other exposed
     * to the foundations toward the street. Its construction, over a
     * century and a half ago, had followed the grading and straightening
     * of the road in that especial vicinity; for Benefit Street--at
     * first called Back Street--was laid out as a lane winding amongst
     * the graveyards of the first settlers, and straightened only when
     * the removal of the bodies to the North Burial Ground made it
     * decently possible to cut through the old family plots.
     */
    fn construct(&self) -> bool;
}

/// Whatever
impl TheShunnedHouse for OmNomNomy {
    fn dingy_house(&self, _unkempt_yard: int) {
    }

    fn construct(&self) -> bool {
        fail!();
    }
}
