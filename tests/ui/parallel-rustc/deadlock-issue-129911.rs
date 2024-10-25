//@ parallel-front-end
//@ compile-flags: -Z threads=16

fn main() {
    type KooArc = Frc<
        //~^ ERROR
        {
            {
                {
                    {};
                }
                type Frc = Frc<{}>::Arc;;
                //~^ ERROR
                //~| ERROR
            }
            type Frc = Frc<
                //~^ ERROR
                //~| ERROR
                {
                    {
                        {
                            {};
                        }
                        type Frc = Frc<{}>::Arc;;
                        //~^ ERROR
                        //~| ERROR
                    }
                    type Frc = Frc<
                        //~^ ERROR
                        //~| ERROR
                        {
                            {
                                {
                                    {};
                                }
                                type Frc = Frc<{}>::Arc;;
                                //~^ ERROR
                                //~| ERROR
                            }
                            type Frc = Frc<
                                //~^ ERROR
                                //~| ERROR
                                {
                                    {
                                        {
                                            {};
                                        }
                                        type Frc = Frc<{}>::Arc;;
                                        //~^ ERROR
                                        //~| ERROR
                                    }
                                    type Frc = Frc<
                                        //~^ ERROR
                                        //~| ERROR
                                        {
                                            {
                                                {
                                                    {
                                                        {};
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                    //~^ ERROR
                                                    //~| ERROR
                                                };
                                            }
                                            type Frc = Frc<
                                                //~^ ERROR
                                                //~| ERROR
                                                {
                                                    {
                                                        {
                                                            {};
                                                        };
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                    //~^ ERROR
                                                    //~| ERROR
                                                },
                                            >::Arc;;
                                        },
                                    >::Arc;;
                                },
                            >::Arc;;
                        },
                    >::Arc;;
                },
            >::Arc;;
        },
    >::Arc;
}
