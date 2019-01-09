// error-pattern:expected `[`, found `vec`
mod blade_runner {
    #vec[doc(
        brief = "Blade Runner is probably the best movie ever",
        desc = "I like that in the world of Blade Runner it is always
                raining, and that it's always night time. And Aliens
                was also a really good movie.

                Alien 3 was crap though."
    )]
}
