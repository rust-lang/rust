// Test for #129911, deadlock detected as we're unable to find a query cycle to break

fn main() {
    type KooArc = Frc<
    //~^ ERROR cannot find type `Frc` in this scope
        {
            {
                {
                    {};
                }
                type Frc = Frc<{}>::Arc;;
                //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                //~| ERROR cycle detected when expanding type alias
            }
            type Frc = Frc<
            //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
            //~| ERROR cycle detected when expanding type alias
                {
                    {
                        {
                            {};
                        }
                        type Frc = Frc<{}>::Arc;;
                        //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                        //~| ERROR cycle detected when expanding type alias
                    }
                    type Frc = Frc<
                    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                    //~| ERROR cycle detected when expanding type alias
                        {
                            {
                                {
                                    {};
                                }
                                type Frc = Frc<{}>::Arc;;
                                //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                //~| ERROR cycle detected when expanding type alias
                            }
                            type Frc = Frc<
                            //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                            //~| ERROR cycle detected when expanding type alias
                                {
                                    {
                                        {
                                            {};
                                        }
                                        type Frc = Frc<{}>::Arc;;
                                        //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                        //~| ERROR cycle detected when expanding type alias
                                    }
                                    type Frc = Frc<
                                    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                    //~| ERROR cycle detected when expanding type alias
                                        {
                                            {
                                                {
                                                    {
                                                        {};
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                                    //~| ERROR cycle detected when expanding type alias
                                                };
                                            }
                                            type Frc = Frc<
                                            //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                            //~| ERROR cycle detected when expanding type alias
                                                {
                                                    {
                                                        {
                                                            {};
                                                        };
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied
                                                    //~| ERROR cycle detected when expanding type alias
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
