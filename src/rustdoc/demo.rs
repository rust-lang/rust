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

#[doc = "The type of things that produce omnomnom"]
enum omnomnomy {
    #[doc = "Delicious sugar cookies"]
    cookie,
    #[doc = "It's pizza"]
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

#[doc(
    brief = "Bored",
    desc = "

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nec
    molestie nisl. Duis massa risus, pharetra a scelerisque a,
    molestie eu velit. Donec mattis ligula at ante imperdiet ut
    dapibus mauris malesuada. Sed gravida nisi a metus elementum sit
    amet hendrerit dolor bibendum. Aenean sit amet neque massa, sed
    tempus tortor. Sed ut lobortis enim. Proin a mauris quis nunc
    fermentum ultrices eget a erat. Mauris in lectus vitae metus
    sodales auctor. Morbi nunc quam, ultricies at venenatis non,
    pellentesque ac dui.

    Quisque vitae est id eros placerat laoreet sit amet eu
    nisi. Curabitur suscipit neque porttitor est euismod
    lacinia. Curabitur non quam vitae ipsum adipiscing
    condimentum. Mauris ut ante eget metus sollicitudin
    blandit. Aliquam erat volutpat. Morbi sed nisl mauris. Nulla
    facilisi. Phasellus at mollis ipsum. Maecenas sed convallis
    sapien. Nullam in ligula turpis. Pellentesque a neque augue. Sed
    eget ante feugiat tortor congue auctor ac quis ante. Proin
    condimentum lacinia tincidunt.

")]
resource bored(bored: bool) {
    log(error, bored);
}