// no-reformat

#[doc = "

    A demonstration module

    Contains documentation in various forms that rustdoc understands,
    for testing purposes. It doesn't surve any functional
    purpose. This here, for instance, is just some filler text.

    FIXME (1654): It would be nice if we could run some automated
    tests on this file

"];

#[doc = "The base price of a muffin on a non-holiday"]
const price_of_a_muffin: float = 70f;

type waitress = {
    hair_color: str
};

enum omnomnomy {
    cookie,
    pizza_pie([uint])
}

fn take_my_order_please(
    _waitress: waitress,
    _order: [omnomnomy]
) -> uint {

    #[doc(
        desc = "OMG would you take my order already?",
        args(_waitress = "The waitress that you want to bother",
             _order = "The order vector. It should be filled with food."),
        return = "The price of the order, including tax",
        failure = "This function is full of fail"
    )];

    fail;
}

mod fortress_of_solitude {
    #[doc(
        brief = "Superman's vacation home",
        desc = "

        The fortress of solitude is located in the Arctic and it is
        cold. What you may not know about the fortress of solitude
        though is that it contains two separate bowling alleys. One of
        them features bumper-bowling and is kind of lame.

        Really, it's pretty cool.

    ")];

}

mod blade_runner {
    #[doc(
        brief = "Blade Runner is probably the best movie ever",
        desc = "I like that in the world of Blade Runner it is always
                raining, and that it's always night time. And Aliens
                was also a really good movie.

                Alien 3 was crap though."
    )];
}
